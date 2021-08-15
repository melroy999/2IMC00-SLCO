/**
 */
package poosl.tests;

import junit.textui.TestRunner;

import poosl.IncomingMessage;
import poosl.PooslFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Incoming Message</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class IncomingMessageTest extends MessageTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(IncomingMessageTest.class);
	}

	/**
	 * Constructs a new Incoming Message test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public IncomingMessageTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Incoming Message test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected IncomingMessage getFixture() {
		return (IncomingMessage)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PooslFactory.eINSTANCE.createIncomingMessage());
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

} //IncomingMessageTest
