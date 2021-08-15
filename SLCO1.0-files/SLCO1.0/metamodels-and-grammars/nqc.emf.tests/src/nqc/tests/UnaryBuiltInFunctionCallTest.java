/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.NqcFactory;
import nqc.UnaryBuiltInFunctionCall;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Unary Built In Function Call</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class UnaryBuiltInFunctionCallTest extends BuiltInFunctionCallTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(UnaryBuiltInFunctionCallTest.class);
	}

	/**
	 * Constructs a new Unary Built In Function Call test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public UnaryBuiltInFunctionCallTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Unary Built In Function Call test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected UnaryBuiltInFunctionCall getFixture() {
		return (UnaryBuiltInFunctionCall)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createUnaryBuiltInFunctionCall());
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

} //UnaryBuiltInFunctionCallTest
