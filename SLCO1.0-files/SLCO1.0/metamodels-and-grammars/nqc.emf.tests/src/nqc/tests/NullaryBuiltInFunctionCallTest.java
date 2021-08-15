/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.NullaryBuiltInFunctionCall;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Nullary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class NullaryBuiltInFunctionCallTest extends BuiltInFunctionCallTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(NullaryBuiltInFunctionCallTest.class);
	}

	/**
	 * Constructs a new Nullary Built In Function Call test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NullaryBuiltInFunctionCallTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Nullary Built In Function Call test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected NullaryBuiltInFunctionCall getFixture() {
		return (NullaryBuiltInFunctionCall)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createNullaryBuiltInFunctionCall());
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

} //NullaryBuiltInFunctionCallTest
