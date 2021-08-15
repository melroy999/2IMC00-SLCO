/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.TernaryBuiltInFunctionCall;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Ternary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class TernaryBuiltInFunctionCallTest extends BuiltInFunctionCallTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(TernaryBuiltInFunctionCallTest.class);
	}

	/**
	 * Constructs a new Ternary Built In Function Call test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TernaryBuiltInFunctionCallTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Ternary Built In Function Call test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected TernaryBuiltInFunctionCall getFixture() {
		return (TernaryBuiltInFunctionCall)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createTernaryBuiltInFunctionCall());
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#tearDown()
	 * @generated
	 */
	@Override
	protected void tearDown() throws Exception {
		setFixture(null);
	}

} //TernaryBuiltInFunctionCallTest
